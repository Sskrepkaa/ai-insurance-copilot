"""FastAPI app for the AI Insurance Copilot backend.

Implements the full fixed API contract (00_architecture.md): the two read
endpoints (GET /claims, GET /claims/{claim_id}) plus the LangGraph-backed
POST /chat and POST /resume endpoints for the tool-calling + HITL agent
defined in agent.py.

NOTE for the next agent (AI Judge): /chat and /resume build their response
as a plain `dict` (`response = {...}`) before returning it - just add a
`response["judge_score"] = ...` line in each handler once judge_response()
exists. Nothing here needs to change shape-wise to accommodate that.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent
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

    response = {
        "type": "answer",
        "text": agent.extract_final_text(result["messages"]),
    }
    # judge-agent slots in here later, e.g.:
    #   response["judge_score"] = judge_response(...)
    return response


@app.post("/resume")
def resume(req: ResumeRequest):
    """Resume a paused agent thread with the human expert's decision."""
    config = {"configurable": {"thread_id": req.thread_id}}
    result = agent.graph.invoke(
        agent.Command(resume={"approved": req.approved, "reason": req.reason}),
        config=config,
    )

    response = {
        "type": "answer",
        "text": agent.extract_final_text(result["messages"]),
    }
    # judge-agent slots in here later, e.g.:
    #   response["judge_score"] = judge_response(...)
    return response


# --- How to run ---------------------------------------------------------
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# python3 seed_data.py
# uvicorn main:app --reload --port 8000
#
# Verify: open http://localhost:8000/claims in a browser -> JSON list of claims.

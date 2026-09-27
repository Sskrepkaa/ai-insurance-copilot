"""FastAPI app for the AI Insurance Copilot backend.

Only implements the two read endpoints from the fixed API contract
(00_architecture.md). /chat and /resume are intentionally NOT implemented
here - they will be added by the LangGraph agent, which imports tools.py.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


# --- How to run ---------------------------------------------------------
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# python3 seed_data.py
# uvicorn main:app --reload --port 8000
#
# Verify: open http://localhost:8000/claims in a browser -> JSON list of claims.

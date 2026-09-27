"""Tool functions used (later) by the LangGraph agent.

IMPORTANT: this module must be importable with NO side effects - no DB
connections are opened, no tables are created, at import time. Every
function opens its own short-lived connection when called.
"""

from db import get_connection


def get_claim_details(claim_id: str) -> dict:
    """Return full claim data for `claim_id`, or {"error": "not found"}."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, client, amount, status, type, description FROM claims WHERE id = ?",
            (claim_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return {"error": "not found"}

    return dict(row)


def check_fraud_risk(claim_id: str) -> dict:
    """Compute a simple demo fraud risk score for `claim_id`.

    risk = min(amount / 20000, 1.0)
    + 0.3 if the client has a past "fraud_detected" entry in customer_history
    capped at 1.0 overall.

    Returns {"risk_score": float, "level": "low"/"medium"/"high"}, or
    {"error": "not found"} if the claim doesn't exist.
    """
    claim = get_claim_details(claim_id)
    if "error" in claim:
        return claim

    risk = min(claim["amount"] / 20000, 1.0)

    conn = get_connection()
    try:
        fraud_row = conn.execute(
            "SELECT 1 FROM customer_history WHERE client = ? AND outcome = 'fraud_detected' LIMIT 1",
            (claim["client"],),
        ).fetchone()
    finally:
        conn.close()

    if fraud_row is not None:
        risk += 0.3

    risk = min(risk, 1.0)

    if risk < 0.34:
        level = "low"
    elif risk < 0.67:
        level = "medium"
    else:
        level = "high"

    return {"risk_score": round(risk, 4), "level": level}


def get_customer_history(client: str) -> list:
    """Return the list of past claims/outcomes for `client` from customer_history."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, client, claim_id, date, outcome FROM customer_history "
            "WHERE client = ? ORDER BY date",
            (client,),
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]

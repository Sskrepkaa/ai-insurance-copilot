const API_BASE = "http://localhost:8000";

async function parseJsonOrThrow(res, action) {
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body?.detail ? ` - ${body.detail}` : "";
    } catch {
      // response had no JSON body, ignore
    }
    throw new Error(`${action} failed (${res.status})${detail}`);
  }
  return res.json();
}

export async function fetchClaims() {
  const res = await fetch(`${API_BASE}/claims`);
  return parseJsonOrThrow(res, "Loading claims");
}

export async function fetchClaimDetails(claimId) {
  const res = await fetch(`${API_BASE}/claims/${claimId}`);
  return parseJsonOrThrow(res, "Loading claim details");
}

export async function sendMessage(claimId, message) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ claim_id: claimId, message }),
  });
  return parseJsonOrThrow(res, "Sending message");
}

export async function resumeDecision(threadId, approved, reason) {
  const res = await fetch(`${API_BASE}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, approved, reason }),
  });
  return parseJsonOrThrow(res, "Submitting decision");
}

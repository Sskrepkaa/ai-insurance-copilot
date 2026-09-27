---
name: frontend-agent
description: Builds the React + Vite dashboard for the AI Insurance Copilot project — claim list, per-claim chat panel, judge pass/fail indicator, and Approve/Reject HITL buttons, consuming the fixed backend API contract (GET /claims, GET /claims/{id}, POST /chat, POST /resume) on http://localhost:8000. Does not touch backend code — only needs the API contract. Can run independently of backend/langgraph/judge agents once the contract is fixed.
tools: Read, Write, Edit, Bash, Glob, Grep
isolation: worktree
---

You build the React frontend for the "AI Copilot for insurance support" demo. Read `00_architecture.md` at the repo root for the full contract; the summary below is authoritative for your part of the work. The backend runs at `http://localhost:8000` with these endpoints:
- `GET /claims` → `[{"id", "client", "amount", "status", "type"}, ...]`
- `GET /claims/{claim_id}` → `{"id", "client", "amount", "status", "type", "description", "fraud_risk"}`
- `POST /chat` body `{"claim_id", "message"}` → either `{"type": "answer", "text", "judge_score": {"groundedness", "relevance", "verdict"}}` or `{"type": "interrupt", "question", "thread_id"}`
- `POST /resume` body `{"thread_id", "approved", "reason"}` → `{"type": "answer", "text", "judge_score": {...}}`

## Stack
Vite + React (plain JSX, no TypeScript, to keep it simple). Styling: plain CSS or Tailwind via CDN (skip build config).

## Layout — 2 columns

### Left column: claim list
- Load `GET /claims` on mount
- Each card shows: id, client, amount, type, a colored status badge (pending=yellow, approved=green, rejected=red)
- Clicking a card selects it, loads details via `GET /claims/{id}`, opens the right column

### Right column: chat with the agent (appears once a claim is selected)
- Header: "Claim {id} - {client}"
- Message history (simple list, user vs agent styled differently)
- Text input + "Send" button → `POST /chat {claim_id, message}`
- After the agent responds:
  - if `response.type === "answer"`: show the answer text plus a small judge indicator dot — green if `judge_score.verdict === "pass"`, red if `"fail"`, with a tooltip showing the groundedness/relevance numbers
  - if `response.type === "interrupt"`: show a visually distinct block (e.g. yellow border/background) with `response.question` and two buttons "✅ Approve" / "❌ Reject". On click → `POST /resume {thread_id: response.thread_id, approved: true/false, reason: "expert decision via dashboard"}`, then show the final agent answer

## Files
```
src/
  App.jsx           - overall layout, selected-claim state
  ClaimList.jsx      - left column, list of cards
  ClaimCard.jsx       - one claim card
  ChatPanel.jsx       - right column, all chat + HITL logic
  api.js              - fetchClaims(), fetchClaimDetails(id), sendMessage(claimId, message), resumeDecision(threadId, approved, reason)
```

## api.js — reference shape
```javascript
const API_BASE = "http://localhost:8000";

export async function sendMessage(claimId, message) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({claim_id: claimId, message}),
  });
  return res.json();
}
```

## UX details that matter
- While waiting for `/chat` to respond, show a simple spinner / "Agent is thinking..."
- The judge indicator should be noticeable but not distracting — a small dot next to the agent's message, not a separate large block
- The HITL block with the buttons must look visually distinct (e.g. yellow border/background) so it's obviously "not a normal answer, needs a human decision"

## Run
```
npm create vite@latest frontend -- --template react
cd frontend && npm install
npm run dev
```
Open `http://localhost:5173` and confirm the claim list loads from the backend (backend must already be running on port 8000 for this check — if it isn't, note that in your report rather than treating it as your bug).

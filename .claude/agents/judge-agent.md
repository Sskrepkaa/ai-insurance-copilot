---
name: judge-agent
description: Adds the AI Judge module (backend/judge.py) to the AI Insurance Copilot backend — a separate Groq LLM call that scores a chat answer for groundedness/relevance against tool_outputs, and wires judge_score into the POST /chat response. Requires langgraph-agent to have already added the /chat endpoint in backend/main.py. Do not use for building the chat/HITL graph itself or for frontend work.
tools: Read, Write, Edit, Bash, Glob, Grep
isolation: worktree
---

You add a `judge.py` module to `backend/` that checks agent answers for "groundedness" (did the agent invent facts not present in the tool outputs). Read `00_architecture.md` at the repo root for the full project contract before starting.

## Task
Create `backend/judge.py`:
```python
def judge_response(question: str, answer: str, tool_outputs: list[dict]) -> dict:
    """
    Makes a separate LLM call (Groq, same llama-3.3-70b-versatile) acting as judge.
    The judge receives: the support employee's question, the agent's final answer,
    and all the data returned by tools (get_claim_details, check_fraud_risk,
    get_customer_history).
    """
```

Judge prompt (adapt wording but keep the substance):
```
You are an AI judge for quality control of an insurance assistant's answers.
You are given: the user's question, the agent's final answer, and the factual
data the agent obtained from tools (tool_outputs).

Check TWO criteria and score each from 0.0 to 1.0:
1. groundedness - does the answer contain ONLY facts present in tool_outputs,
   with no invented numbers/facts absent from there
2. relevance - does the text actually answer the question asked

Return ONLY valid JSON, no explanation:
{"groundedness": 0.0-1.0, "relevance": 0.0-1.0, "verdict": "pass"|"fail", "reason": "brief why"}

verdict = "fail" if groundedness < 0.7 OR relevance < 0.7

Question: {question}
Agent answer: {answer}
Tool data: {tool_outputs}
```

Parse the judge's JSON response with `json.loads`, and handle parse failures by returning `{"groundedness": None, "verdict": "error"}`.

## Integration into main.py
In the `POST /chat` endpoint, AFTER the agent's final answer is obtained:
1. Collect `tool_outputs` — all `ToolMessage` entries from the graph's message history for this call
2. Call `judge_response(question=message, answer=<agent's text>, tool_outputs=tool_outputs)`
3. Add the result to the response: `{"type": "answer", "text": ..., "judge_score": {...}}`

Do the same for `POST /resume`'s final answer, since that response also needs a `judge_score` per the API contract.

## Why this matters (keep in mind while implementing)
The AI Judge is a SEPARATE LLM call, not the same one that answered the user. In real systems, the judge is sometimes a more capable/expensive model than the main agent, because its job is strictly quality control, not generation. It can later be calibrated against real human-expert ratings by collecting (answer, expert score) pairs and comparing — out of scope for this task, just keep `judge_response` self-contained enough that this is easy to add later.

## Verification
Call `/chat` with a question that has clear factual grounding (e.g. about claim CLM-1001) and confirm `judge_score.groundedness` is close to 1.0. Then try prompting the agent to "make up an amount if you don't know it" and confirm the groundedness score drops.
